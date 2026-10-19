import unittest
from unittest.mock import MagicMock, patch
from db.message_feedback import correct_message

class MessageFeedbackTests(unittest.TestCase):
    def run_case(self, rows, label='anger', user_id=4):
        connection=MagicMock(); cursor=connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect=rows
        return patch('db.message_feedback.get_connection',return_value=connection), cursor

    def test_foreign_message_is_rejected(self):
        connection,_=self.run_case([(5,'sad','sadness')])
        with patch('db.message_feedback.ensure_schema'),connection,patch('db.message_feedback.save_emotion_feedback') as save:
            with self.assertRaises(PermissionError): correct_message('room','message',4,'TEST','anger')
            save.assert_not_called()

    def test_duplicate_does_not_add_training_example(self):
        connection,_=self.run_case([(4,'sad','sadness'),('anger',)])
        with patch('db.message_feedback.ensure_schema'),connection,patch('db.message_feedback.save_emotion_feedback') as save:
            self.assertEqual(correct_message('room','message',4,'TEST','anger'),'anger')
            save.assert_not_called()

    def test_correction_uses_database_text_and_original_label(self):
        connection,_=self.run_case([(4,'sad','sadness'),None])
        with patch('db.message_feedback.ensure_schema'),connection,patch('db.message_feedback.save_emotion_feedback') as save:
            correct_message('room','message',4,'TEST','anger')
            self.assertEqual(save.call_args.args[0],'TEST')
            self.assertEqual(save.call_args.args[2:],('sadness','anger'))

if __name__=='__main__':unittest.main()
